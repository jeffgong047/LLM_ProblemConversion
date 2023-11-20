import leetcode
import leetcode.auth

import re
import json
import markdownify
from time import sleep

class leetcode_wrapper():
    def __init__(self) -> None:
        '''
        Set up configure
        '''
        # Get the next two values from your browser cookies
        leetcode_session = 'xxxx' # replace it
        csrf_token = 'xxx' # replace it

        # Experimental: Or CSRF token can be obtained automatically
        csrf_token = leetcode.auth.get_csrf_cookie(leetcode_session)
        configuration = leetcode.Configuration()

        configuration.api_key["x-csrftoken"] = csrf_token
        configuration.api_key["csrftoken"] = csrf_token
        configuration.api_key["LEETCODE_SESSION"] = leetcode_session
        configuration.api_key["Referer"] = "https://leetcode.com"
        configuration.debug = False

        self.api_instance = leetcode.DefaultApi(leetcode.ApiClient(configuration))

    def get_question_detail(self, titleSlug):
        """
        retrieve question detail given title slug
        """
        graphql_request = leetcode.GraphqlQuery(
        query="""
                query getQuestionDetail($titleSlug: String!) {
                question(titleSlug: $titleSlug) {
                    questionId
                    questionFrontendId
                    boundTopicId
                    title
                    content
                    translatedTitle
                    isPaidOnly
                    difficulty
                    likes
                    dislikes
                    isLiked
                    similarQuestions
                    contributors {
                    username
                    profileUrl
                    avatarUrl
                    __typename
                    }
                    langToValidPlayground
                    topicTags {
                    name
                    slug
                    translatedName
                    __typename
                    }
                    companyTagStats
                    codeSnippets {
                    lang
                    langSlug
                    code
                    __typename
                    }
                    stats
                    codeDefinition
                    hints
                    solution {
                    id
                    canSeeDetail
                    __typename
                    }
                    status
                    sampleTestCase
                    enableRunCode
                    metaData
                    translatedContent
                    judgerAvailable
                    judgeType
                    mysqlSchemas
                    enableTestMode
                    envInfo
                    __typename
                }
                }
            """,
            variables=leetcode.GraphqlQueryGetQuestionDetailVariables(title_slug=titleSlug),
            operation_name="getQuestionDetail",
        )

        response = self.api_instance.graphql_post(body=graphql_request)
        # Select some import parts of it, can be modified
        selected_response = {
            'content': re.sub(r'\n+', '\n', markdownify.markdownify(response.data.question.content, heading_style="ATX")),
            'difficulty': response.data.question.difficulty,
            'hints': response.data.question.hints,
            'question_frontend_id': response.data.question.question_frontend_id,
            'question_id': response.data.question.question_id,
            'title': response.data.question.title,
            'title_slug': response.data.question.title_slug,
            'topic_tags': response.data.question.topic_tags,
        }

        return selected_response


    def submission(self, code, question_id, titleSlug, language='python'):
        # Send code to leetcode for evaluation
        # question_id would be provided in the retrived question detail
        submission = leetcode.Submission(
            judge_type="large", typed_code=code, question_id=question_id, test_mode=False, lang=language
        )

        submission_id = self.api_instance.problems_problem_submit_post(
            problem=titleSlug, body=submission
        )

        sleep(5)  # FIXME: should probably be a busy-waiting loop

        submission_result = self.api_instance.submissions_detail_id_check_get(
            id=submission_id.submission_id
        )

        del submission_result['task_name']
        del submission_result['finished']

        results = leetcode.SubmissionResult(**submission_result)
        
        return results


if __name__ == '__main__':
    # Below is an example of sending code to 'two-sum'
    titleSlug = "two-sum"
    code = """
class Solution:
    def twoSum(self, nums, target):
        print("stdout")
        return [1]
    """

    leetcode_call = leetcode_wrapper()
    question_detail = leetcode_call.get_question_detail(titleSlug)

    print(question_detail)

    leetcode_call.submission(code, 1, titleSlug)

    """Sample Output
    {'code_output': '[1]',
    'compare_result': '000000000000000000000000000000000000000000000000000000000000',
    'elapsed_time': 54,
    'expected_output': '[0,1]',
    'full_runtime_error': None,
    'input': '[2,7,11,15]\n9',
    'input_formatted': '[2,7,11,15], 9',
    'lang': 'python',
    'last_testcase': '[2,7,11,15]\n9',
    'memory': 13876000,
    'memory_percentile': None,
    'pretty_lang': 'Python',
    'question_id': 1,
    'run_success': True,
    'runtime_error': None,
    'runtime_percentile': None,
    'state': 'SUCCESS',
    'status_code': 11,
    'status_memory': 'N/A',
    'status_msg': 'Wrong Answer',
    'status_runtime': 'N/A',
    'std_output': 'stdout\n',
    'submission_id': '1094883250',
    'task_finish_time': 1699497149127,
    'total_correct': 0,
    'total_testcases': 60}
    """